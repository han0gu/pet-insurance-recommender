from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자는 서면 등을 발송한 때에 그 발<br>송 사실을 회사에 지체없이 알려야 합니다.<br>\uf000 계약자가 청약을 철회한 '
 '때에는 회사는 청약의 철회를 접수한 날부터 3영업일 이내<br>에 납입한 보험료를 돌려드리며, 보험료 반환이 늦어진 기간에 대하여는 이 '
 '계약의<br>보험계약대출이율을 연단위 복리로 계산한 금액을 더하여 지급합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000162',
              'chunk_char_len': 187,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
