from langchain_core.documents import Document

chunk = Document(
    page_content=('정신건강의학과 의료기관에서 실시되어져야<br>하며, 자격을 갖춘 임상심리전문가가 시행하고 작성하여야 한다.<br>차) 정신행동장해 진단 '
 '전문의는 정신건강의학과 전문의를 말한다.<br>카) 정신행동장해는 뇌의 기능 및 결손을 입증할 수 있는 뇌자기공명촬<br>영, '
 '뇌전산화촬영, 뇌파 등 객관적 근거를 기초로 평가한다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001665',
              'chunk_char_len': 176,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
