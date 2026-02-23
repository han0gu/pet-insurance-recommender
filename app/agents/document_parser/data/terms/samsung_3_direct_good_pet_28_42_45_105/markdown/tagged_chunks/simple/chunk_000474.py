from langchain_core.documents import Document

chunk = Document(
    page_content=('당시의 이 특별약관의 계약자적립액 및 미경과보험료를 지급하고, 부활(효력회복)\n'
 '일 이후부터 납입한 이 특별약관의 보험료를 돌려 드립니다.# 제 5조 (특별약관의 소멸)- ① 회사가 제1조(보험금의 지급사유) '
 '제1항에서 정한 사망보험금을 지급한 때에는 그 손\n'
 '- 해보상의 원인이 생긴 때부터 이 특별약관은 소멸되며 그 때부터 효력이 없습니다. 이\n'
 '- 경우 회사는 이 특별약관의 해약환급금을 지급하지 않습니다.\n'
 '- ② 보험증권에 기재된 반려견이 보험기간 중에 이 특별약관에서 보장하지 않는 사유로'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000474',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
