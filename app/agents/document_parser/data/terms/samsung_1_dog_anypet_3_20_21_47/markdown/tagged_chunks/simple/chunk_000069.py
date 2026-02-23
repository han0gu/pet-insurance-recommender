from langchain_core.documents import Document

chunk = Document(
    page_content=('- 력회복)을 청약한 경우에는 계약이 해지된 날부터 7일이 되는 날에 특별부활(효력회복) 됩니다.\n'
 '- 15 -당신에게 좋은보험 삼성화재④ 피보험자는 통지를 받은 날부터 15일 이내에 제1항의 절차를 이행할 수 있습니다.제6관 계약의 '
 '해지 및 보험료의 환급 등# 제26조(계약의 해지)- ① 계약자는 손해가 발생하기 전에는 언제든지 계약을 해지할 수 있습니다. 다만, '
 '타인을 위한 계약의\n'
 '- 경우에는 계약자는 그 타인의 동의를 얻거나 보험증권을 소지한 경우에 한하여 계약을 해지할 수\n'
 '- 있습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000069',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
