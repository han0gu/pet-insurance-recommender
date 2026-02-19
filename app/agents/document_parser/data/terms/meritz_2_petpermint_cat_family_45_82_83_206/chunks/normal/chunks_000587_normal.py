from langchain_core.documents import Document

chunk = Document(
    page_content=('【 별첨 】특정질병 분류표(반려묘)\n'
 '보험계약을 체결할 때 반려동물의 건강상태가 회사가 정한 기준에 적합 하지 않은 경우 또는 보험계약을 체결한 후 계약 전 알릴 의무 위반의 '
 '효과 등으로 보장을 제한할 경우에 한하여 보상하지 않는 질병을 분류 한 표입니다. 보상하지 않는 질병(부담보 질병)은 회사가 정한 기준에 '
 '따라 직접 관련이 있는 특정질병으로 제한합니다.\n'
 '구 분 | 특정질병 | 분류코드 | 항목명\n'
 '1 | 뒷다리 근골격계 질환 | AEB003 | 뒷다리의 골육종\n'
 'AEA004 | 기타 근골격 계통의 양성 신생물(뒷다리)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 169},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000587',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
