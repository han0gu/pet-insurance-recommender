from langchain_core.documents import Document

chunk = Document(
    page_content=('약에 부가하여 이루어집니다.\n'
 '② 제1조(적용대상)의 보험계약이 해지 또는 기타 사유에 의하여 효력이 없게 된 경우에\n'
 '는 이 특별약관은 더 이상 효력이 없습니다.- \n'
 '# 제3조 (지정대리청구인의 지정)① 계약자는 보험계약에서 정한 보험금을 직접 청구할 수 없는 특별한 사정이 있을 경우\n'
 '를 대비하여 계약을 체결할 때 또는 계약체결 이후 다음 각 호의 어느 하나에 해당하\n'
 '는 자 중에서 보험금의 청구대리인(2인 이내에서 지정하되, 2인 지정시 대표대리인을'),
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
 'indexing': {'chunk_id': 'chunk_000545',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
