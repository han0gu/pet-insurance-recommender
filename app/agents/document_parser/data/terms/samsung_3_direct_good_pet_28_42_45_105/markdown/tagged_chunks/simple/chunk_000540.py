from langchain_core.documents import Document

chunk = Document(
    page_content=('- 사 안내가 계약자에게 도달하지 못한 경우 포함)에는 갱신일에 갱신일 현재의 약관\n'
 '- 등으로 갱신됩니다. 다만, 계약자는 갱신일 현재의 약관 등에 대해 갱신일로부터 90\n'
 '- 일 이내에 그 계약을 취소할 수 있으며, 이 경우 회사는 갱신일 이후 계약자가 납입\n'
 '- 한 해당 갱신계약의 보험료를 돌려 드립니다.'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000540',
              'chunk_char_len': 171,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
