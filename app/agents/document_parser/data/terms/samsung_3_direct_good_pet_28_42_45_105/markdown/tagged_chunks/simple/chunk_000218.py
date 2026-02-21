from langchain_core.documents import Document

chunk = Document(
    page_content=('관의 보험계약대출이율이 변경되는 경우, 변경된 시점부터 변경된 이율을 적용합니다.③ 제1항 및 제2항에 의한 보험료의 자동대출납입 기간은 '
 '최초 자동대출납입일부터 1년을 한도로 하며 그 이후의 기간에 대한 보험료의 자동대출납입을 위해서는 제1항에- 54 -54 / 181# '
 '따라 재신청을 하여야 합니다.- ④ 보험료의 자동대출납입이 행하여진 경우에도 자동대출납입 전 납입최고(독촉)기간이\n'
 '- 끝나는 날의 다음날부터 1개월 이내에 계약자가 계약의 해지를 청구한 때에는 회사는'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000218',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
