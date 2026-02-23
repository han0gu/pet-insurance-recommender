from langchain_core.documents import Document

chunk = Document(
    page_content=('검사# 【전산화단층영상(CT)】X선을 이용하여 반려동물의 횡단면상의 영상을 획득하여\n'
 '진단에 이용하는 검사# 【내시경】내장장기 또는 체강(體腔) 내부를 직접 볼 수 있게 만든\n'
 '의료기구# 제5조(특별약관의 소멸)이 특별약관에서 정한 보상하는 손해가 더 이상 발생할 수\n'
 '없는 경우에는 이 특별약관은 그 때부터 소멸되며, 이 경우\n'
 '회사는「보험료 및 해약환급금 산출방법서」에서 정한 이\n'
 '특별약관의 그 때까지 적립한 계약자적립액 및 미경과보험\n'
 '료를 지급합니다.# 제6조(준용규정)이 특별약관에서 정하지 않은 사항은「반려동물 비용손해'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000408',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
