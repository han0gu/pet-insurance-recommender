from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, " 천식", "중증 천식", "급성 천식", "만성 천식"과 같은 용어로 표현되는 천식 (J45)은 이 계약에서 보장하지 '
 "않습니다.</td></tr></tbody></table><h1 id='215' "
 "style='font-size:14px'>제4조(특별약관의</h1><br><p id='216' "
 "data-category='paragraph' style='font-size:14px'>소멸)</p><br><p id='217' "
 "data-category='list' style='font-size:14px'>\uf000 회사가 제1조(보험금의"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000665',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
