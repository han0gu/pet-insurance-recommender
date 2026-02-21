from langchain_core.documents import Document

chunk = Document(
    page_content=('. 수탁기관 위탁비용 영수증 및 동물관리위탁업자가 제공하는 계약서(위탁관리<br>업소 등록번호, 업소명 및 주소, 전화번호, 위탁관리동물 '
 "종류, 품종, 나이,<br>서비스 기간, 비용 등 포함)</p><br><p id='134' data-category='paragraph' "
 "style='font-size:14px'>7"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001295',
              'chunk_char_len': 178,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
