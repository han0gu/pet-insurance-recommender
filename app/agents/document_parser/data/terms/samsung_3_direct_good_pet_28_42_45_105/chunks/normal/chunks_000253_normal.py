from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[보험계약대출이율]\n'
 '해당 보험상품의 약관에 따라 계약자가 대출을 받을 경우, 회사가 정하는 대출이율이며, 이 특별약 관의 보험계약대출이율이 변경되는 경우, '
 '변경된 시점부터 변경된 이율을 적용합니다.\n'
 '③ 제1항 및 제2항에 의한 보험료의 자동대출납입 기간은 최초 자동대출납입일부터 1년\n'
 '을 한도로 하며 그 이후의 기간에 대한 보험료의 자동대출납입을 위해서는 제1항에'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 54},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000253',
              'chunk_char_len': 210,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
