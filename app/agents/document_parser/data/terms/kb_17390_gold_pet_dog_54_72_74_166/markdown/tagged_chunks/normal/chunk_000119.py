from langchain_core.documents import Document

chunk = Document(
    page_content=('전에 무효임을 알았거나 알 수 있었음에도 보험료를 반환하지 않은 경우에는 보험료\n'
 '를 납입한 날의 다음날부터 반환일까지의 기간에 대하여 회사는 이 계약의 보험계약- 대출이율을 연단위 복리로 계산한 금액을 더하여 돌려 '
 '드립니다.\n'
 '- 1. 타인의 사망을 보험금 지급사유로 하는 계약에서 계약을 체결할 때까지 피보험\n'
 '- 자의 서면(전자서명법 제2조 제2호에 따른 전자서명이 있는 경우로서 상법 시\n'
 '- 행령 제44조의2에 정하는 바에 따라 본인 확인 및 위조·변조 방지에 대한 신'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000119',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
