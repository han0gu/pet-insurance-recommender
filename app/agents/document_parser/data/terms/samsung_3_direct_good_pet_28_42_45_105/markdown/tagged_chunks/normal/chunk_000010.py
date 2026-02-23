from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1항에 따라 장해지급률이 결정되었으나 그 이후 보장받을 수 있는 기간(계약의 효\n'
 '- 력이 없어진 경우에는 보험기간이 10년 이상인 계약은 상해 발생일부터 2년 이내로\n'
 '- 하고, 보험기간이 10년 미만인 계약은 상해 발생일부터 1년 이내)에 장해상태가 더 악\n'
 '- 화된 때에는 그 악화된 장해상태를 기준으로 장해지급률을 결정합니다.\n'
 '- ③ 장해분류표에 해당되지 않는 후유장해는 피보험자의 직업, 연령, 신분 또는 성별 등에\n'
 '- 관계없이 신체의 장해정도에 따라 장해분류표의 구분에 준하여 지급액을 결정합니다.'),
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000010',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
