from langchain_core.documents import Document

chunk = Document(
    page_content=('되는 질병</td><td>분류번호</td></tr><tr><td>탄광부 진폐증</td><td>J60 '
 '공</td></tr><tr><td>석면 및 기타 광섬유에 의한 진폐증</td><td>J61 통</td></tr><tr><td>실리카를 '
 '함유한 먼지에 의한 진폐증</td><td>J62 사항</td></tr><tr><td>기타 무기물먼지에 의한 '
 '진폐증</td><td>J63</td></tr><tr><td>상세불명의 진폐증</td><td>J64</td></tr><tr><td>결핵과 '
 '연관된'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001793',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
