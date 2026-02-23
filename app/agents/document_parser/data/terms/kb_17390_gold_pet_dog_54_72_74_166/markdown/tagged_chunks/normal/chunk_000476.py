from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에 따른 이자를 지급하지 않습니다.\n'
 '|  |\n'
 '| --- |\n'
 '| 용 어 풀 이 정당한 사유 의무의 이행을 당사자에게 기대하는 것이 무리라고 할 만한 사정이 있을 때 특 |\n'
 '# (책임을 물을 만한 기대가능성이 없을 때)를 말하며, 가령 천재지변, 전쟁,사변 등으로 인해 이행이 불가능한 경우 등이 있습니다. '
 '약\n'
 '관\n'
 '\uf000 회사는 제5항의 서면조사에 대한 동의 요청시 조사목적, 사용처 등을 명시하고설명합니다.별제6조(보험금의 분담)\n'
 '\uf000 회사는 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약 상'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000476',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
