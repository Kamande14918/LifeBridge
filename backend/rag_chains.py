from typing import List, Dict, Any
import re
import json
from life_bridge_logger import logger

# Resilient imports: LangChain refactors and packaging differences across
# versions can move classes between modules. Try known import paths and
# fall back to lightweight local shims where appropriate so the app can
# still run for development or limited environments.
try:
    # preferred (newer) import path
    from langchain.prompts import PromptTemplate
except Exception:
    try:
        # older LangChain or alternate packaging
        from langchain import PromptTemplate
    except Exception:
        # minimal fallback PromptTemplate used only for formatting the
        # prompt string when LangChain is not installed or has moved APIs.
        class PromptTemplate:
            def __init__(self, template: str, input_variables: List[str]):
                self.template = template
                self.input_variables = input_variables

            def format(self, **kwargs) -> str:
                # Basic Python format-based substitution. Keep simple to
                # avoid introducing dependencies; callers using advanced
                # PromptTemplate features will still need LangChain.
                return self.template.format(**kwargs)

try:
    from langchain.chains import ConversationalRetrievalChain
except Exception:
    # If LangChain is missing, raise a clear error later when used.
    ConversationalRetrievalChain = None

try:
    from langchain.schema import Document
except Exception:
    try:
        from langchain.docstore.document import Document
    except Exception:
        # Provide a minimal Document shim for local index/testing.
        class Document:
            def __init__(self, page_content: str, metadata: dict = None):
                self.page_content = page_content
                self.metadata = metadata or {}

try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline
except Exception:
    # Keep names defined so import-time errors don't break module load; real
    # functionality will require installing these packages.
    FAISS = None
    HuggingFaceEmbeddings = None
    HuggingFacePipeline = None
from transformers import pipeline
from config import settings
from life_bridge_logger import logger

def load_vectorstore() -> FAISS:
    logger.info(f"Loading FAISS index from {settings.INDEX_DIR}")
    if HuggingFaceEmbeddings is None or FAISS is None:
        raise RuntimeError(
            "Required vectorstore/embeddings packages not installed. "
            "Install `langchain-community` or `langchain-huggingface` and `faiss` per the README."
        )
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    vs = FAISS.load_local(settings.INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vs

def build_llm():
    logger.info(f"Loading generator: {settings.GENERATION_MODEL}")
    # Use low temperature and deterministic decoding to reduce hallucinations.
    # `do_sample=False` ensures greedy decoding where possible.
    # Ensure the LLM pipeline adapter is available
    if HuggingFacePipeline is None:
        raise RuntimeError(
            "Required LLM adapter `HuggingFacePipeline` not available. "
            "Install `langchain-huggingface` or update LangChain to a compatible release."
        )

    gen_pipe = pipeline(
        "text2text-generation",
        model=settings.GENERATION_MODEL,
        max_new_tokens=settings.MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.0,
    )
    return HuggingFacePipeline(pipeline=gen_pipe)

def build_prompt():
    # Require the model to return EXACTLY one valid JSON object with the
    # required fields. This avoids fragile free-text parsing and makes the
    # system deterministic when the model follows the instruction.
    # If the context does not contain an answer, return:
    #   {"insufficient_context": true, "fallback": "short safety fallback..."}
    # and nothing else (no extra text, headings, or explanation).
    template = (
        "You are a First Aid assistant for LifeBridge First Aid in Kenya.\n"
        "Use ONLY the provided context (do NOT use prior model knowledge).\n"
        "Return EXACTLY one valid JSON object and nothing else. Do not output any other text.\n"
        "The JSON object MUST include the following keys: \n"
        "  - steps: array of short strings (imperative steps).\n"
        "  - warnings: array of short strings.\n"
        "  - escalate: array of strings (when to call emergency services).\n"
        "  - contacts: array of contact strings (may be empty).\n"
        "  - visual_guide: string URL or null.\n"
        "Optional keys: sources (array of source titles), confidence (number 0.0-1.0).\n"
        "If the context does NOT contain an answer, return EXACTLY: \n"
        "  {{\"insufficient_context\": true, \"fallback\": \"short safety fallback...\"}}\n"
        "Context:\n{context}\n\nChat history:\n{chat_history}\n\nUser question:\n{question}\n"
    )
    return PromptTemplate(template=template, input_variables=["context", "question", "chat_history"])


def _extract_primary_actions_from_doc(d) -> List[str]:
    """Extract primary-action steps from a Document.
    Look in metadata for 'primary_action' then scan page_content for
    a 'Primary Action:' marker. Return a list of non-empty step strings.
    """
    out: List[str] = []
    try:
        md = d.metadata if isinstance(d.metadata, dict) else getattr(d, 'metadata', {})
        if isinstance(md, dict):
            pa = md.get('primary_action') or md.get('Primary Action')
        else:
            pa = getattr(md, 'primary_action', None) or getattr(md, 'Primary Action', None)

        if pa and isinstance(pa, str) and pa.strip():
            parts = [p.strip() for p in re.split(r'[\.;]\s*', pa) if p.strip()]
            out.extend(parts)

        if not out:
            pc = getattr(d, 'page_content', '')
            if isinstance(pc, str) and pc:
                # Find the Primary Action line and take the first logical segment
                m = re.search(r'Primary Action\s*:\s*(.*)', pc, re.I)
                if m:
                    pa_text = m.group(1).strip()
                    # Only take up to the first newline (the Primary Action is typically on one line)
                    pa_text = pa_text.splitlines()[0].strip()
                    parts = [p.strip() for p in re.split(r'[\.;]\s*', pa_text) if p.strip()]
                    out.extend(parts)
                # If contacts/title are missing in metadata, try extract them from page_content
                try:
                    if not out:
                        # also attempt to find a line like 'Emergency Contact: ...'
                        m2 = re.search(r'Emergency Contact\s*:\s*(.*)', pc, re.I)
                        if m2:
                            ec = m2.group(1).splitlines()[0].strip()
                            if ec:
                                out.append(f"Contact info found: {ec}")
                except Exception:
                    pass
    except Exception:
        pass
    return out

def chain_with_memory(vectorstore, llm, memory):
    # If LangChain's ConversationalRetrievalChain isn't available in this
    # environment, provide a lightweight fallback that performs a simple
    # retrieval from the vectorstore and calls the LLM pipeline directly.
    # This avoids the opaque `'NoneType' object has no attribute 'from_llm'`
    # errors and ensures the API can still return answers for testing.
    use_fallback = False
    if ConversationalRetrievalChain is None:
        logger.warning(
            "ConversationalRetrievalChain not available; using simple fallback retrieval chain. "
            "Install full LangChain packages for production functionality."
        )
        use_fallback = True

    prompt = build_prompt()
    # Extra safety: ensure the imported object exposes the expected API
    if not use_fallback:
        if not getattr(ConversationalRetrievalChain, 'from_llm', None):
            raise ImportError(
                "LangChain's `ConversationalRetrievalChain.from_llm` is not available. "
                "This usually means your installed LangChain (or related packages) "
                "is missing or incompatible. Install the project's dependencies:\n\n"
                "  .\\venv\\Scripts\\Activate.ps1\n"
                "  pip install -r requirements.txt\n\n"
                "Or install the components directly:\n"
                "  pip install langchain langchain-huggingface langchain-community sentence-transformers faiss-cpu\n"
            )
        retriever = vectorstore.as_retriever(search_kwargs={"k": settings.TOP_K})
        # ConversationalRetrievalChain injects memory chat history and retrieved docs into prompt
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            # Ensure memory knows which output to persist when the chain returns
            # multiple keys (e.g. 'answer' and 'source_documents'). Setting
            # output_key to 'answer' tells LangChain to store the textual
            # answer into the conversation memory.
            output_key="answer",
            return_source_documents=True,
            verbose=False,
        )
    else:
        # Simple fallback implementation:
        # 1) retrieve top-k docs from the vectorstore
        # 2) construct a context string and format the prompt
        # 3) call the LLM pipeline directly and return a similar result shape
        def _retrieve_docs(query: str, k: int = settings.TOP_K):
            # Try common retrieval APIs
            try:
                if hasattr(vectorstore, "as_retriever"):
                    retr = vectorstore.as_retriever(search_kwargs={"k": k})
                    return retr.get_relevant_documents(query)
            except Exception:
                pass
            try:
                if hasattr(vectorstore, "similarity_search"):
                    return vectorstore.similarity_search(query, k=k)
            except Exception:
                pass
            raise RuntimeError("Vectorstore does not expose a recognisable retrieval API")

        class _SimpleFallbackChain:
            def __init__(self, vectorstore, llm, memory):
                self.vectorstore = vectorstore
                self.llm = llm
                self.memory = memory

            def invoke(self, inputs: Dict[str, Any]):
                question = inputs.get("question") or inputs.get("query") or ""
                docs = _retrieve_docs(question, k=settings.TOP_K)
                # Build context from retrieved docs
                context_parts = []
                for d in docs:
                    title = d.metadata.get("title") if isinstance(d.metadata, dict) else getattr(d.metadata, 'title', '')
                    context_parts.append(f"Source: {title}\n{getattr(d, 'page_content', str(d))}")
                context = "\n\n".join(context_parts)
                chat_history = ""  # We don't integrate complex history in the fallback
                prompt_text = prompt.format(context=context, question=question, chat_history=chat_history)

                # Call LLM: support both HuggingFacePipeline wrapper or raw transformers pipeline
                generated = None
                if hasattr(self.llm, 'pipeline'):
                    pipe = self.llm.pipeline
                    out = pipe(prompt_text, max_new_tokens=settings.MAX_NEW_TOKENS)
                    if isinstance(out, list) and out:
                        generated = out[0].get('generated_text') or out[0].get('text')
                    elif isinstance(out, dict):
                        generated = out.get('generated_text') or out.get('text')
                else:
                    # Try calling callable LLM objects
                    try:
                        out = self.llm(prompt_text)
                        if isinstance(out, dict):
                            generated = out.get('generated_text') or out.get('answer')
                        elif isinstance(out, str):
                            generated = out
                        else:
                            generated = str(out)
                    except Exception as e:
                        raise RuntimeError(f"LLM invocation failed in fallback chain: {e}")

                if generated is None:
                    raise RuntimeError("LLM did not return generated text in fallback chain")

                return {"answer": generated, "source_documents": docs}

        chain = _SimpleFallbackChain(vectorstore, llm, memory)
    # LangChain sometimes attempts to save to memory automatically and may
    # raise if multiple output keys are present. To keep full memory
    # functionality while avoiding ambiguous automatic saves, wrap the
    # chain so we manually control what gets written to memory.

    class _MemoryProxy:
        """A thin proxy that exposes the read methods used by chains but
        no-op's the save_context call to prevent automatic ambiguous writes.
        """
        def __init__(self, real):
            self._real = real

        def load_memory_variables(self, inputs: Dict[str, Any]):
            return self._real.load_memory_variables(inputs)

        def __getattr__(self, name):
            # Delegate other attributes to the real memory (e.g., chat_memory)
            return getattr(self._real, name)

        def save_context(self, *args, **kwargs):
            # Intentionally no-op: we'll save explicitly after invoke
            return None

    class _WrappedChain:
        def __init__(self, inner_chain, real_memory):
            self._chain = inner_chain
            self._real_memory = real_memory
            # Replace the chain's memory with the proxy to prevent auto-save
            if real_memory is not None:
                self._chain.memory = _MemoryProxy(real_memory)
                # Also monkeypatch the real memory's save_context so that if any
                # internal chain tries to call it directly, it will only store
                # the 'answer' field (avoids ambiguous multiple output keys).
                if hasattr(real_memory, 'save_context'):
                    original_save = real_memory.save_context

                    def _safe_save(inputs, outputs):
                        try:
                            # If outputs contains multiple keys, pick 'answer'
                            if isinstance(outputs, dict) and 'answer' in outputs:
                                original_save(inputs, {'answer': outputs.get('answer')})
                            else:
                                # If single key or not a dict, pass through
                                original_save(inputs, outputs)
                        except Exception as e:
                            logger.warning(f"Wrapped memory.save_context failed: {e}")

                    # Replace method
                    try:
                        real_memory.save_context = _safe_save
                    except Exception:
                        # Some memory implementations may not allow assignment; ignore
                        pass

        def invoke(self, inputs: Dict[str, Any]):
            # Call the underlying chain
            result = self._chain.invoke(inputs)
            # After successful invoke, persist only the 'answer' into memory if
            # the real memory supports save_context and wasn't already called.
            try:
                if self._real_memory is not None and hasattr(self._real_memory, 'save_context'):
                    # Call with a single-key output to avoid ambiguity
                    self._real_memory.save_context(inputs, {'answer': result.get('answer')})
            except Exception:
                logger.warning("Failed to save chat memory (non-fatal)")
            return result

    return _WrappedChain(chain, memory)

def parse_answer(text: str):
    # First, try to extract a JSON object from the model output. Newer
    # prompt enforces JSON-only output; if present, parse and map fields
    # directly. If JSON extraction fails, fall back to legacy line parser
    # (handles older model outputs).
    # Improved parser that handles lines that start with numbering (e.g. "1)", "1.")
    # and header lines while preserving any text that follows the marker.
    
    # Try parsing JSON embedded in the model output
    try:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            payload = m.group(0)
            obj = json.loads(payload)
            # Handle explicit insufficient context signal
            if obj.get("insufficient_context") or obj.get("INSUFFICIENT_CONTEXT"):
                return {
                    "steps": [],
                    "warnings": [],
                    "escalate": ["INSUFFICIENT_CONTEXT"],
                    "contacts": [],
                    "visual_guide": None,
                }

            steps = obj.get("steps") or []
            warnings = obj.get("warnings") or []
            escalate = obj.get("escalate") or []
            contacts = obj.get("contacts") or []
            visual = obj.get("visual_guide") if obj.get("visual_guide") is not None else obj.get("visual")

            # Ensure all lists are lists of stripped strings
            def norm_list(x):
                if not x:
                    return []
                if isinstance(x, list):
                    return [str(i).strip() for i in x if str(i).strip()]
                if isinstance(x, str):
                    return [s.strip() for s in x.split("\n") if s.strip()]
                return []

            return {
                "steps": norm_list(steps),
                "warnings": norm_list(warnings),
                "escalate": norm_list(escalate),
                "contacts": norm_list(contacts),
                "visual_guide": visual,
            }
    except Exception:
        # If JSON parsing fails, continue to legacy parsing below
        pass
    steps: List[str] = []
    warnings: List[str] = []
    escalate: List[str] = []
    contacts: List[str] = []
    visual = None
    section = None

    # Common regex to capture numeric list markers and the remainder of the line
    num_re = re.compile(r"^\s*\d+\s*[\)\.:\-]?\s*(.*)$")

    for line in text.splitlines():
        raw = line
        ln = line.strip()
        if not ln:
            continue

        # Detect the explicit INSUFFICIENT_CONTEXT marker and return fallback
        if ln == "INSUFFICIENT_CONTEXT":
            return {
                "steps": [],
                "warnings": [],
                "escalate": ["INSUFFICIENT_CONTEXT"],
                "contacts": [],
                "visual_guide": None,
            }

        low = ln.lower()

        # Visual guide detection (may be embedded anywhere)
        if "visual guide:" in low and "http" in low:
            try:
                visual = raw.split(":", 1)[1].strip()
            except Exception:
                visual = ln.split(":", 1)[1].strip()
            continue

        # Section headers: if a header appears, switch section and if the
        # header line contains trailing text include it in that section.
        if "steps" in low or low.startswith("1"):
            # capture any trailing text after a numeric marker
            m = num_re.match(raw)
            section = "steps"
            if m and m.group(1).strip():
                steps.append(m.group(1).strip())
            # If header had no trailing text, continue to next line
            if not (m and m.group(1).strip()):
                continue
            else:
                continue

        if "do not" in low or low.startswith("2"):
            m = num_re.match(raw)
            section = "warnings"
            if m and m.group(1).strip():
                warnings.append(m.group(1).strip())
            if not (m and m.group(1).strip()):
                continue
            else:
                continue

        if "escalate" in low or "call emergency" in low or low.startswith("3"):
            m = num_re.match(raw)
            section = "escalate"
            if m and m.group(1).strip():
                escalate.append(m.group(1).strip())
            if not (m and m.group(1).strip()):
                continue
            else:
                continue

        if "contacts" in low or low.startswith("4"):
            m = num_re.match(raw)
            section = "contacts"
            if m and m.group(1).strip():
                contacts.append(m.group(1).strip())
            if not (m and m.group(1).strip()):
                continue
            else:
                continue

        # If the line doesn't explicitly start a new section, append it to
        # the currently active section (if any) or otherwise ignore.
        if section == "steps":
            # preserve the line content, strip leading numeric markers if present
            m = num_re.match(raw)
            if m:
                val = m.group(1).strip()
            else:
                val = ln
            if val:
                steps.append(val)
        elif section == "warnings":
            warnings.append(ln)
        elif section == "escalate":
            escalate.append(ln)
        elif section == "contacts":
            contacts.append(ln)

    # If we failed to parse structured sections but there is text, return the
    # whole answer as a single-step fallback (keeps previous behaviour).
    if not steps and not warnings and not escalate and not contacts:
        # Return the raw text as a single step to avoid returning empty items
        raw_text = text.strip()
        return {
            "steps": [raw_text] if raw_text else [],
            "warnings": [],
            "escalate": [],
            "contacts": [],
            "visual_guide": visual,
        }

    return {
        "steps": steps,
        "warnings": warnings,
        "escalate": escalate,
        "contacts": contacts,
        "visual_guide": visual,
    }

def compute_confidence(source_docs: List[Document]) -> float:
    if not source_docs:
        return 0.0
    # Heuristic: more non-empty sources -> higher confidence. If source
    # titles are empty (or all blank), reduce confidence because the chain
    # may be hallucinating sources.
    titles = [getattr(d.metadata, 'title', None) if hasattr(d, 'metadata') else d.metadata.get('title') if isinstance(d.metadata, dict) else None for d in source_docs]
    non_empty = sum(1 for t in titles if t)
    base = min(non_empty / max(settings.TOP_K, 1), 1.0)
    # Scale between 0.2 and 1.0 to avoid presenting overly confident values
    return round(0.2 + 0.8 * base, 3)


def generate_rule_based_fallback(question: str, docs: List = None) -> Dict[str, Any]:
    """Generate a conservative, structured fallback for common first-aid
    scenarios based on simple keyword matching. This is used when
    retrieval confidence is low so callers get actionable guidance.

    The output matches the parsed structure used by the server:
      { steps, warnings, escalate, contacts, visual_guide, sources, confidence }
    """
    q = (question or "").lower()

    # If docs are provided, try to extract useful info from them first
    if docs:
        try:
            # Build sources and contacts from docs metadata
            sources = []
            contacts = []
            visual = None
            steps = []
            for d in docs:
                # metadata may be dict or object
                md = d.metadata if isinstance(d.metadata, dict) else getattr(d, 'metadata', {})
                title = md.get('title') if isinstance(md, dict) else getattr(md, 'title', None)
                if title:
                    sources.append(title)
                c = md.get('contacts') if isinstance(md, dict) else getattr(md, 'contacts', None)
                if c:
                    # contacts stored as ';' separated string in our CSV
                    if isinstance(c, str):
                        contacts.extend([x.strip() for x in c.split(';') if x.strip()])
                    elif isinstance(c, list):
                        contacts.extend(c)
                if not visual:
                    v = md.get('visual') if isinstance(md, dict) else getattr(md, 'visual', None)
                    if v:
                        visual = v
                # Extract primary action steps from metadata or page content
                try:
                    pa_steps = _extract_primary_actions_from_doc(d)
                    if pa_steps:
                        steps.extend(pa_steps)
                except Exception:
                    pass
            if steps:
                return {
                    'steps': steps,
                    'warnings': ["Do not administer medications or perform invasive procedures."],
                    'escalate': ["Call emergency services if signs of severe injury or deterioration."],
                    'contacts': contacts or ["Kenya Red Cross (Ambulance): 1199"],
                    'visual_guide': visual,
                    'sources': sources,
                    'confidence': 0.6,
                }
        except Exception:
            # fall through to default fallback if parsing fails
            pass

    # default generic conservative fallback
    fallback = {
        "steps": [
            "Ensure scene safety and avoid hazards.",
            "Call emergency services immediately.",
            "Provide reassurance and monitor breathing until help arrives."
        ],
        "warnings": [
            "Do not administer medications or perform invasive procedures."
        ],
        "escalate": [
            "If unresponsive or not breathing, begin CPR until help arrives."
        ],
        "contacts": [
            "Kenya Red Cross (Ambulance): 1199",
            "National Emergency / Police: 999 / 112",
            "E-plus Medical Response: 0700 395 395"
        ],
        "visual_guide": None,
        "sources": [],
        "confidence": 0.25,
    }

    # Burns
    if any(w in q for w in ["burn", "scald", "hot water", "heated", "fire"]):
        return {
            "steps": [
                "Ensure scene safety — remove the casualty from the heat source and switch off any flame or electrical supply if safe to do so. [Cooking burn accident (open fire/kitchen)]",
                "Cool the burn immediately with cool running water for at least 10–20 minutes. Do NOT use ice. [St John Ambulance Burns Guide]",
                "Remove rings and watches quickly before swelling. Cover the burn loosely with a clean, non-adhesive dressing. [Kenya Red Cross — Burns]",
                "For large, deep, or circumferential burns or burns to face/hands/feet/genitals/joints, arrange urgent transfer to hospital. [Kenya Red Cross]",
            ],
            "warnings": [
                "Do not apply creams, butter, oil or traditional remedies.",
                "Do not break blisters or remove stuck clothing."
            ],
            "escalate": [
                "Call emergency services immediately for deep/large burns, airway burn, or if the casualty is unresponsive or has breathing difficulty."
            ],
            "contacts": fallback["contacts"],
            "visual_guide": "https://www.sja.org.uk/globalassets/images/first-aid-guides/burns.png",
            "sources": ["Cooking burn accident (open fire/kitchen)", "St John Ambulance — Burns Guide"],
            "confidence": 0.6,
        }

    # Severe bleeding
    if any(w in q for w in ["bleed", "bleeding", "hemorrhage", "blood"]):
        return {
            "steps": [
                "Make the scene safe and wear gloves if available. Apply firm direct pressure to the wound using a clean cloth or dressing. [Severe bleeding guide]",
                "If possible, raise the injured limb above heart level. Do not remove embedded objects. [First Aid Manual]",
                "If bleeding is not controlled, apply a pressure bandage or use a tourniquet only if trained and as a last resort. Call emergency services."
            ],
            "warnings": [
                "Do not remove dressings to check the wound — add more dressing on top if needed.",
                "Do not give the casualty anything to drink if there is severe blood loss."
            ],
            "escalate": [
                "Call emergency services immediately for severe or uncontrolled bleeding."
            ],
            "contacts": fallback["contacts"],
            "visual_guide": "https://www.sja.org.uk/globalassets/images/first-aid-guides/severe-bleeding.png",
            "sources": ["Severe bleeding in road accident (rural highway)", "First Aid Manual"],
            "confidence": 0.65,
        }

    # Choking
    if any(w in q for w in ["choke", "choking", "can't breathe", "airway"]):
        return {
            "steps": [
                "If the casualty can cough forcefully, encourage continued coughing. [Choking guidance]",
                "If they cannot breathe, give 5 back blows between the shoulder blades, then 5 abdominal thrusts (adults) — for infants use chest thrusts. Repeat until the obstruction clears or the casualty becomes unresponsive.",
                "If the casualty becomes unresponsive, start CPR and call emergency services."
            ],
            "warnings": ["Do not perform blind finger sweeps unless you can see an object in the mouth."],
            "escalate": ["Call emergency services immediately if the airway is not cleared or the casualty is deteriorating."],
            "contacts": fallback["contacts"],
            "visual_guide": "https://www.sja.org.uk/globalassets/images/first-aid-guides/choking-child.png",
            "sources": ["Choking child (rural home or school)", "St John Ambulance — Choking"],
            "confidence": 0.6,
        }

    # Snakebite
    if any(w in q for w in ["snake", "viper", "bite"]):
        return {
            "steps": [
                "Keep the person calm and still. Immobilize the bitten limb and keep it lower than heart level. [Kenya Snakebite Guidance]",
                "Do not cut, suck, or apply traditional remedies to the bite. Remove tight clothing and jewelry.",
                "Call emergency services and transport to hospital for antivenom assessment."
            ],
            "warnings": ["Do not apply tourniquets unless instructed by medical personnel."],
            "escalate": ["Seek urgent hospital treatment; call ambulance if progressive swelling, difficulty breathing, or altered consciousness occurs."],
            "contacts": ["Snakebite Hotline EMKF: 0732 429 940"] + fallback["contacts"],
            "visual_guide": None,
            "sources": ["Snake bite (farm field)", "EMKF Snakebite Hotline"],
            "confidence": 0.6,
        }

    # Drowning / not breathing after water rescue
    if any(w in q for w in ["drown", "drowning", "near drowning"]):
        return {
            "steps": [
                "Ensure scene safety and remove the casualty from the water quickly but safely.",
                "Open airway and check breathing — if not breathing, start CPR immediately and call emergency services.",
                "Keep the casualty warm and transport to hospital for observation even if breathing returns."
            ],
            "warnings": ["Do not delay CPR; secondary drowning can still occur — hospital observation is recommended."],
            "escalate": ["Call ambulance immediately for any unresponsive or breathing-compromised casualty after water submersion."],
            "contacts": fallback["contacts"],
            "visual_guide": "https://www.redcross.org.uk/get-involved/teaching-resources/~/media/Images/first-aid/drowning.png",
            "sources": ["Drowning (lake, river, pool)", "Red Cross Drowning Guidance"],
            "confidence": 0.6,
        }

    # Poisoning (ingestion of household chemicals, medicines)
    if any(w in q for w in ["poison", "poisoning", "ingested", "ingest", "chemic", "pesticide"]):
        return {
            "steps": [
                "Remove the casualty from the source of poison to fresh air if safe to do so.",
                "If the poison was swallowed, do NOT induce vomiting unless instructed by a poison control expert.",
                "Try to identify the substance and bring the container to hospital for assessment. Call the Poisons Centre for immediate advice."
            ],
            "warnings": [
                "Do not give activated charcoal or attempt home remedies without professional advice.",
                "If chemical on skin or in eyes, flush with copious water for at least 15 minutes."
            ],
            "escalate": [
                "Call the Poisons Centre or emergency services immediately for ingestion of pesticides, unknown substances, large overdoses, or any deterioration."
            ],
            "contacts": ["Poisons Centre: 0703 618 367"] + fallback["contacts"],
            "visual_guide": None,
            "sources": ["Suspected poisoning (pesticides, medicine)", "Poisons Centre"] ,
            "confidence": 0.6,
        }

    # Complicated childbirth / postpartum haemorrhage
    if any(w in q for w in ["childbirth", "labour", "postpartum", "bleeding after birth", "postpartum hemorrhage"]):
        return {
            "steps": [
                "Call emergency services immediately and arrange urgent transfer to the nearest maternity facility.",
                "If trained: perform uterine massage to encourage contraction and apply firm manual pressure to external bleeding sites.",
                "Encourage breastfeeding if the mother is able, to help stimulate uterine contraction while waiting for help."
            ],
            "warnings": [
                "Do not perform invasive procedures at home unless trained.",
                "Avoid giving the mother anything by mouth if major bleeding is present and she is likely to need surgery."
            ],
            "escalate": [
                "Call ambulance immediately for heavy bleeding after delivery, loss of consciousness, or signs of shock."
            ],
            "contacts": ["Nearest maternity facility; Kenyatta National Hospital" ] + fallback["contacts"],
            "visual_guide": None,
            "sources": ["Complicated childbirth (postpartum hemorrhage, home births)", "WHO Postpartum Hemorrhage Guidance"],
            "confidence": 0.6,
        }

    # Severe asthma attack
    if any(w in q for w in ["asthma", "wheeze", "can't breathe", "breathless", "inhaler"]):
        return {
            "steps": [
                "Help the casualty sit upright and stay calm. Assist them to use their reliever inhaler (usually a blue inhaler) — 4 puffs via spacer if available, one puff at a time.",
                "If no improvement after 5 minutes, repeat the inhaler treatment and call emergency services.",
                "If the casualty becomes exhausted, drowsy or stops talking in sentences, treat as life-threatening and call ambulance immediately."
            ],
            "warnings": [
                "Do not give sedating medications or oral steroids unless prescribed and advised by medical personnel."
            ],
            "escalate": [
                "Call emergency services immediately for severe or deteriorating asthma not responding to inhaler treatment."
            ],
            "contacts": fallback["contacts"],
            "visual_guide": "https://www.asthma.org.uk/advice/triggers/first-aid-asthma.png",
            "sources": ["Severe asthma attack (unable to breathe)", "Asthma First Aid Guidance"],
            "confidence": 0.6,
        }

    # Open / suspected fracture
    if any(w in q for w in ["fracture", "broken", "bone", "open fracture", "compound fracture"]):
        return {
            "steps": [
                "Stop any obvious bleeding with firm direct pressure using a clean dressing.",
                "Immobilize the limb in the position found using splints or rolled-up clothing to prevent movement. Avoid moving the casualty unnecessarily.",
                "Cover any open wounds with sterile dressing and seek urgent transport to hospital for orthopaedic assessment."
            ],
            "warnings": [
                "Do not attempt to realign bones or push exposed bone back in — cover with a sterile dressing and get to hospital."
            ],
            "escalate": [
                "Call ambulance for suspected open fractures, limb deformity with severe pain, or signs of compromised circulation."
            ],
            "contacts": fallback["contacts"],
            "visual_guide": "https://www.sja.org.uk/globalassets/images/first-aid-guides/broken-bone.png",
            "sources": ["Open fracture (compound fracture from fall/accident)", "First Aid Manual"],
            "confidence": 0.6,
        }

    # Seizure
    if any(w in q for w in ["seizure", "fit", "convulsion", "epilepsy"]):
        return {
            "steps": [
                "Protect the person from injury by removing nearby hard or sharp objects; cushion their head.",
                "Do not restrain or place anything in their mouth. Time the seizure — if it lasts more than 5 minutes, call emergency services.",
                "Once the seizure ends, place the person in the recovery position if breathing and monitor until help arrives."
            ],
            "warnings": [
                "Do not give oral medication or fluids during a seizure."
            ],
            "escalate": [
                "Call ambulance if the seizure lasts longer than 5 minutes, repeats without recovery, or if the person is injured or pregnant."
            ],
            "contacts": fallback["contacts"],
            "visual_guide": "https://www.sja.org.uk/globalassets/images/first-aid-guides/epilepsy.png",
            "sources": ["Seizure (first-time or recurrent convulsions)", "Epilepsy First Aid Guidance"],
            "confidence": 0.6,
        }

    # Default: return the conservative fallback
    return fallback
