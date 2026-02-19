from langchain_core.documents import Document

chunk = Document(
    page_content=('<유의사항>\n'
 '계약자 또는 피보험자는 상해보험계약을 맺은 후 피보험자가 직업 또는 직무를 변경(자가용운전자가 영업용운전자로 직업 또는 직무 변경 '
 '포함)하거나 이륜자동차 또는 원동기장치 자전거를 계속적으로 사용하게 된 경우에는 즉시 회사에 알려야 합니다. 그러지 않을 경우 보험사고가 '
 '발생한 경우에도\n'
 '보험금 지급이 제한될 수 있습니다.\n'
 '원동기장치 자전거는 전동킥보드, 전동이륜평행차, 전동기의 동력만으로 움직일 수 있는 자전거 등'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 49},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000196',
              'chunk_char_len': 238,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
