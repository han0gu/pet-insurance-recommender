from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 3-1. 반려견 의료비(치과및구강 질환포함)(수술당일제외, 검사비포함)(재가입형) 특별약관의 보험금이 1일당 보상한도 액과 '
 '동일한 경우에 한하여 보상합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 77},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000460',
              'chunk_char_len': 92,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
