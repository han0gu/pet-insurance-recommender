from langchain_core.documents import Document

chunk = Document(
    page_content=('하지 않은 경우 또는 보험계약을 체결한 후 계약 전 알릴 의무 위반의\n'
 '효과 등으로 보장을 제한할 경우에 한하여 보상하지 않는 질병을 분류\n'
 '한 표입니다. 보상하지 않는 질병(부담보 질병)은 회사가 정한 기준에\n'
 '따라 직접 관련이 있는 특정질병으로 제한합니다.| 구 분 | 특정질병 | 분류코드 | 항목명 |\n'
 '| --- | --- | --- | --- |\n'
 '| 1 | 뒷다리 근골격계 질환 | AEB003 | 뒷다리의 골육종 |\n'
 '| 1 | 뒷다리 근골격계 질환 | AEA004 | 기타 근골격 계통의 양성 신생물(뒷다리) |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000468',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
