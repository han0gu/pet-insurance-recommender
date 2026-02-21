from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 피보험자가 사실대로 알리는 것을 방해한 경우, 계약자 또는 피보험자에게 사실\n'
 '- 대로 알리지 않게 하였거나 부실한 사항을 알릴 것을 권유했을 때. 다만, 보험설계\n'
 '- 사 등의 행위가 없었다 하더라도 계약자 또는 피보험자가 사실대로 알리지 않거나\n'
 '부실한 사항을 알렸다고 인정되는 경우에는 특별약관을 해지할 수 있습니다.③ 제1항에 따라 특별약관을 해지하였을 때에는 '
 '제35조(해약환급금)제1항에 따른 해약환# 급금을 계약자에게 지급합니다.- ④ 제1항 제1호에 의한 특별약관의 해지가 보험금 지급사유 '
 '발생 후에 이루어진 경우에'),
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000173',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
