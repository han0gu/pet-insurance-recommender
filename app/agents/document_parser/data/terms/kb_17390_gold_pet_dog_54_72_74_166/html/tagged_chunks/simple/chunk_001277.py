from langchain_core.documents import Document

chunk = Document(
    page_content=('. 질<br>\uf000 제1항의 반려동물 위탁비용은 위탁1일당 이 특별약관의 보험가입금액을 한도로 병<br>합니다.</p><h1 '
 "id='111' style='font-size:14px'>제2조(보험금 지급에 관한 세부규정)</h1><br><p id='112' "
 "data-category='paragraph' style='font-size:14px'>\uf000 제1조(보험금의 지급사유) 제1항의 "
 '반려동물 위탁비용은 같은 질병의 치료를 목 상<br>적으로 2회 이상 입원한 경우 이를 1회 입원으로 보아 각 입원일수를 더합니다'),
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
 'indexing': {'chunk_id': 'chunk_001277',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
