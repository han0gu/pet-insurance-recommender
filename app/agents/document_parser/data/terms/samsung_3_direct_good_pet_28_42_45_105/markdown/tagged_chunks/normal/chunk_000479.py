from langchain_core.documents import Document

chunk = Document(
    page_content=('를 입은 사람(법인인 경우에는 그 이사 또는 법인의 업무를 집행하는 그 밖의 기관\n'
 ')을 말합니다.# ② 보상 관련 용어- 1. 배상책임 : 보험기간 중에 발생된 보험사고로 인하여 타인에게 입힌 손해에 대한\n'
 '- 법률상의 책임을 말합니다.\n'
 '- 2. 보상한도액 : 회사와 계약자간에 약정한 금액으로 피보험자가 법률상의 배상책임을\n'
 '- 부담함으로써 입은 손해 중 보험금의 지급한도에 따라 회사가 책임지는 금액의 최\n'
 '- 대 한도를 말합니다.\n'
 '- 3. 자기부담금 : 보험사고로 인하여 발생한 손해에 대하여 계약자 또는 피보험자가 부'),
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
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000479',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
