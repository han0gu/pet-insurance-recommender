from langchain_core.documents import Document

chunk = Document(
    page_content=('일반상해로 사고가 발생한 후 보험금을 청구하였으나 보험금이 약정한 보험금보다 적게 지급되었\n'
 '습니다.# 제15조 (알릴 의무 위반의 효과)① 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생여부에 관계없이 이 계약을 해# '
 '반하고 그 의무가 중요한 사항에 해당하는 경우2. 뚜렷한 위험의 증가와 관련된 제14조(상해보험계약 후 알릴 의무) 제1항에서 정한\n'
 '계약 후 알릴 의무를 계약자 또는 피보험자의 고의 또는 중대한 과실로 이행하지\n'
 '않았을 때② 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 회사는 계약을'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000045',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
