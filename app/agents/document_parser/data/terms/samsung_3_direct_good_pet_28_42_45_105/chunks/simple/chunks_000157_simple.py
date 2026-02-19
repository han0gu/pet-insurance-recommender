from langchain_core.documents import Document

chunk = Document(
    page_content=('⑦ (재가입형) 특별약관 재가입 관련 용어\n'
 '1. 최초계약 : 최초로 체결되는 계약을 말합니다. 2 재가있게야 · OI ㅂ허이 사업방법 서에 서 저하 TULLOI 정차에 MPL '
 '재가인되 게야으'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 45},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000157',
              'chunk_char_len': 106,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
