from langchain_core.documents import Document

chunk = Document(
    page_content=('- 항에 정한 해약환급금은 없거나 최초가입시 안내한 금액보다 적어질 수 있습니다.\n'
 '- ④ 회사는 경과기간별 해약환급금에 관한 표를 계약자에게 제공하여 드립니다.\n'
 '- ⑤ 제30조의2(위법계약의 해지)에 따라 위법계약이 해지되는 경우 회사가 적립한 해지\n'
 '- 당시의 계약자적립액 및 미경과보험료를 반환하여 드립니다.\n'
 '# 제34조 (보험계약대출)① 계약자는 이 계약의 해약환급금 범위 내에서 회사가 정한 방법에 따라 대출(이하 「보\n'
 '험계약대출」 이라 합니다)을 받을 수 있습니다. 그러나 순수보장성보험 등 보험상품의'),
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
 'indexing': {'chunk_id': 'chunk_000113',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
