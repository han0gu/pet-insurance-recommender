from langchain_core.documents import Document

chunk = Document(
    page_content=('. ⑤ 회사가 지급할 제1항에서 정한 의료비보험금(수술당일)은 보험증권에 기재된 자기부 담금을 차감한 후 보상비율을 곱한 금액이며 '
 '보험증권에 기재된 1일당 보상한도액을 한도로 합니다.(자기부담급은 수술 당일 의료비에서 차감합니다.)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 81},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000502',
              'chunk_char_len': 129,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
