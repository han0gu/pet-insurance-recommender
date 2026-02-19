from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자가 서면 이외에 인터넷 또는 전화(음성녹음) 등으로 자동대출납입을 신청할 경 우 회사는 자동대출납입 신청내역을 서면, '
 '전화(음성녹음) 또는 전자문서(SMS포함) 등으로 계약자에게 알려 드립니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 54},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000250',
              'chunk_char_len': 117,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
