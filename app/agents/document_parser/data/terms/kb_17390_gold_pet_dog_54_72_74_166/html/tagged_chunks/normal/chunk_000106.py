from langchain_core.documents import Document

chunk = Document(
    page_content=('↓" data-coord="top-left:(186,115); bottom-right:(708,363)" /></figure><br><p '
 "id='139' data-category='paragraph' style='font-size:14px'>계약변경 완료</p><p "
 "id='140' data-category='list' style='font-size:14px'>\uf000 회사는 제2항에 따라 "
 '계약내용을 변경할 때 위험이 감소된 경우에는 보험료를 감<br>액하고, 이후 기간 보장을 위한 재원인 계약자적립액 등의 차이로 인하여'),
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
 'indexing': {'chunk_id': 'chunk_000106',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
