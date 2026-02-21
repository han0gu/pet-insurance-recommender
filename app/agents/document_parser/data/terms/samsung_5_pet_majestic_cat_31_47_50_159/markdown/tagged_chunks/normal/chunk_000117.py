from langchain_core.documents import Document

chunk = Document(
    page_content=('따른 납입최고(독촉) 등을 실시할 것\n'
 '4.전자적 상품설명장치에 안내의 속도와 음량을 조절할 수 있는 기능을 갖출 것\n'
 '5.제3호 및 제4호의 내용에 관한 사항을 계약자에게 안내할 것⑦ 제1항에 따라 계약이 해지된 경우에는 제36조(해약환급금) 제1항에 '
 '따른 해약환급금# 을 계약자에게 지급합니다.# 제31조 (보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))- ① 제30조(보험료의 '
 '납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)에 따라 계약이'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000117',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
