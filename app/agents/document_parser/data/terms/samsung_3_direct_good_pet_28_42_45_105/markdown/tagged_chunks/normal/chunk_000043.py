from langchain_core.documents import Document

chunk = Document(
    page_content=('을 안 날부터 1개월 이내에 계약자 또는 피보험자에게 제4항에 따라 보장됨을 통보하\n'
 '고 이에 따라 보험금을 지급합니다.# <유의사항>계약자 또는 피보험자는 상해보험계약을 맺은 후 피보험자가 직업 또는 직무를 '
 '변경(자가용운전자가\n'
 '영업용운전자로 직업 또는 직무 변경 포함)하거나 이륜자동차 또는 원동기장치 자전거를 계속적으로\n'
 '사용하게 된 경우에는 즉시 회사에 알려야 합니다. 그러지 않을 경우 보험사고가 발생한 경우에도\n'
 '보험금 지급이 제한될 수 있습니다.'),
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
 'indexing': {'chunk_id': 'chunk_000043',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
