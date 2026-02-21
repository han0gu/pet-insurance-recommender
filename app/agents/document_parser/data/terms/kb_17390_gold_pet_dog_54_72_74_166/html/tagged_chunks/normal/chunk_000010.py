from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:18px'>- 54 -</p><table id='5' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>가</td></tr><tr><td>부 설 "
 '명 한국표준질병․사인분류 부호 체계 질병의 원인과 증상 두 가지 모두에 관한 정보를 포함하는 진단을 위해 아래 두 가지 분류부호가 '
 '사용됩니다'),
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
 'indexing': {'chunk_id': 'chunk_000010',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
