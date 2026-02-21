from langchain_core.documents import Document

chunk = Document(
    page_content=('동의를 한 피보험자는 계약의 효력이 유지되는 기간에는 언제든지 서면동의를 장래를향하여 철회할 수 있으며, 서면동의 철회로 계약이 해지되어 '
 '회사가 지급하여야 할 해약환급금이 있을 때에는 제35조(해약환급금) 제1항에 따른 해약환급금을 계약자에게\n'
 '지급합니다.# 제 32조의2 (위법계약의 해지)① 계약자는 「금융소비자 보호에 관한 법률」제47조 및 관련규정이 정하는 바에 따라\n'
 '계약체결에 대한 회사의 법위반사항이 있는 경우 계약체결일부터 5년 이내의 범위에\n'
 '서 계약자가 위반사항을 안 날부터 1년 이내에 계약해지요구서에 증빙서류를 첨부하'),
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
 'indexing': {'chunk_id': 'chunk_000236',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
