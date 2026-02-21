from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 보험증권에 기재된 반려견이 보험기간 중에 이 특별약관에서 보장하지 않는 사유로\n'
 '- 사망하였을 경우에는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 회사\n'
 '- 가 적립한 사망당시 이 특별약관의 계약자적립액 및 미경과보험료를 계약자에게 지급\n'
 '- 하고, 이 특별약관은 더 이상 효력이 없습니다.\n'
 '제6조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))부활(효력회복)되는 특별약관의 보장개시는 3-1. 반려견 '
 '의료비(치과및구강질환포함)(수'),
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000475',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
