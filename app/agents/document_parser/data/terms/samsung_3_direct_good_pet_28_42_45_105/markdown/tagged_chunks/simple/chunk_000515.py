from langchain_core.documents import Document

chunk = Document(
    page_content=('우체국, 신협, 새마을금고 등이 공제계약을 취급합니다.- 91 -② 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 '
 '제1항에 의한\n'
 '지급보험금 결정에는 영향을 미치지 않습니다.# 제5조 (특별약관의 소멸)피보험자 또는 보험증권에 기재된 반려견이 보험기간 중에 사망하였을 '
 '경우에는 "보험료\n'
 '및 해약환급금 산출방법서"에서 정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의\n'
 '계약자적립액 및 미경과보험료를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없'),
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
 'indexing': {'chunk_id': 'chunk_000515',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
