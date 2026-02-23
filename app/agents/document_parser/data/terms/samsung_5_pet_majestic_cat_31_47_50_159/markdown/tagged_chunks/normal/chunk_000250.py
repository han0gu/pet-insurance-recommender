from langchain_core.documents import Document

chunk = Document(
    page_content=('⑦ 제1항에 따라 계약이 해지된 경우에는 제35조(해약환급금)에서 정한 해약환급금을 계\n'
 '약자에게 지급합니다.# 제 30조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))- ① 제29조(보험료의 납입이 연체되는 '
 '경우 납입최고(독촉)와 특별약관의 해지)에 따라 계\n'
 '- 약이 해지되었으나 해약환급금을 받지 않은 경우(보험계약대출 등에 따라 해약환급금\n'
 '- 이 차감되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를 포함합니다) 계약자\n'
 '- 는 해지된 날부터 3년 이내에 회사가 정한 절차에 따라 계약의 부활(효력회복)을 청약'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000250',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
