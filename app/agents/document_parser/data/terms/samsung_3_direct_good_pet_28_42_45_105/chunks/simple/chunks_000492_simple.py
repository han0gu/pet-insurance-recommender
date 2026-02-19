from langchain_core.documents import Document

chunk = Document(
    page_content=('제 7조 (특별약관의 소멸)\n'
 '① 보험증권에 기재된 반려견이 보험기간 중에 사망하여 보험의 목적에 대해 이 특별약 관에서 정한 보험금 지급사유가 더이상 발생할 수 없는 '
 '경우에는 “보험료 및 해약환 급금 산출방법서”에 정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의 계약 자적립액 및 미경과보험료를 '
 '계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없 습니다.\n'
 '제 8조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 80},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000492',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
