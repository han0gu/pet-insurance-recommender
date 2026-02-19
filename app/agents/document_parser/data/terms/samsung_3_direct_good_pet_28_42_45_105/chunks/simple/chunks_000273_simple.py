from langchain_core.documents import Document

chunk = Document(
    page_content=('향하여 철회할 수 있으며, 서면동의 철회로 계약이 해지되어 회사가 지급하여야 할 해\n'
 '약환급금이 있을 때에는 제35조(해약환급금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.\n'
 '제 32조의2 (위법계약의 해지)\n'
 '① 계약자는 「금융소비자 보호에 관한 법률」제47조 및 관련규정이 정하는 바에 따라 계약체결에 대한 회사의 법위반사항이 있는 경우 '
 '계약체결일부터 5년 이내의 범위에 서 계약자가 위반사항을 안 날부터 1년 이내에 계약해지요구서에 증빙서류를 첨부하 여 위법계약의 해지를 '
 '요구할 수 있습니다.\n'
 '<용어풀이>\n'
 '[위법계약]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 56},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000273',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
