from langchain_core.documents import Document

chunk = Document(
    page_content=('배상책임보장 특별약관\n'
 '제1조(보상하는 손해)\n'
 '① 회사는 피보험자가 국내(북한지역 제외)에서 보험기간 중에 보험증권에 기재된 피보험자의 가입동 물의 행위에 기인하는 우연한 사고(이하 '
 '"사고"라 합니다.)로 인하여 피해자의 신체에 장해(상해, 질병 및 그로 인한 사망을 말합니다.)를 입히거나 피해자 소유의 동물에 손해를 '
 '입혀 법률상의 배 상책임을 부담함으로써 입은 아래의 손해를 이 약관에 따라 보상하여 드립니다.\n'
 '1. 피보험자가 피해자에게 지급할 책임을 지는 법률상의 손해배상금 2. 계약자 또는 피보험자가 지출한 아래의 비용'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 25},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000132',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
