from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 제9조(보험금 등의 청구)에서 정한 서류를 접수한 때에는 접수증을 드리고 휴 대전화 문자메시지 또는 전자우편 등으로 송부하며, '
 '그 서류를 접수한 날부터 3영업 일 이내에 보험금을 지급하거나 보험료의 납입을 면제합니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 47},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000169',
              'chunk_char_len': 126,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
