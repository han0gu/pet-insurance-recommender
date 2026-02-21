from langchain_core.documents import Document

chunk = Document(
    page_content=('5.제3호 및 제4호의 내용에 관한 사항을 계약자에게 안내할 것⑦ 제1항에 따라 특별약관이 해지된 경우에는 이 특별약관의 해약환급금을 '
 '계약자에게\n'
 '지급합니다.# 제22조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))① 제21조(보험료의 납입이 연체되는 경우 '
 '납입최고(독촉)와 특별약관의 해지)에 따라 특\n'
 '별약관이 해지되었으나 해약환급금을 받지 않은 경우(보험계약대출 등에 따라 해약환'),
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
 'indexing': {'chunk_id': 'chunk_000518',
              'chunk_char_len': 222,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
