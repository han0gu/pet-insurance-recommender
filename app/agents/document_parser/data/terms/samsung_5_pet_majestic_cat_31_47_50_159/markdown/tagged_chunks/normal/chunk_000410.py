from langchain_core.documents import Document

chunk = Document(
    page_content=('또는 국외의 의료관련법에서 정한 의료기관에서 발급한 것이어야 합니다.# 제 5조 (특별약관의 소멸)피보험자가 보험기간 중에 사망하였을 '
 '경우에는 "보험료 및 해약환급금 산출방법서"에서\n'
 '정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의 계약자적립액 및 미경과보험료\n'
 '를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없습니다.- 89 -# 2-2. 반려동물 양육자금Ⅱ 특별약관# 제1관 일반사항- '
 '① 제2관 개별사항에서 정하지 않은 사항은 특별약관의 일반사항을 적용합니다. 단, 특별'),
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
 'indexing': {'chunk_id': 'chunk_000410',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
