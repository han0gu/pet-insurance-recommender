from langchain_core.documents import Document

chunk = Document(
    page_content=('중 수술을 한 날의 경우</td></tr></thead><tbody><tr><td '
 'rowspan="2">실속형</td><td>입원</td><td>1일당 10만원 한도 1일당</td><td>150만원 '
 '한도</td><td>1,000만원</td></tr><tr><td>통원</td><td>1일당 10만원 한도 1일당</td><td>150만원 '
 '한도</td><td>1,000만원 상</td></tr><tr><td '
 'rowspan="2">기본형Ⅰ</td><td>입원</td><td>1일당 15만원 한도 1일당</td><td>200만원'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000938',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
