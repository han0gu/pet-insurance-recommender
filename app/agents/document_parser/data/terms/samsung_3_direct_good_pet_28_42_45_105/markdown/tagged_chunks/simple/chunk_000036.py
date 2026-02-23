from langchain_core.documents import Document

chunk = Document(
    page_content=('알았거나 중대한 과실로 인하여 알지 못한 때에는 그러하지 아니하다.# [상법 제651조의2(서면에 의한 질문의 효력)]\n'
 '보험자가 서면으로 질문한 사항은 중요한 사항으로 추정한다.# 제14조 (상해보험계약 후 알릴 의무)① 계약자 또는 피보험자는 보험기간 '
 '중에 피보험자에게 다음 각 호의 변경이 발생한 경\n'
 '우에는 우편, 전화, 방문 등의 방법으로 지체없이 회사에 알려야 합니다.# 1. 보험증권 등에 기재된 직업 또는 직무의 변경- 가. '
 '현재의 직업 또는 직무가 변경된 경우\n'
 '- 나. 직업이 없는 자가 취직한 경우'),
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
 'indexing': {'chunk_id': 'chunk_000036',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
