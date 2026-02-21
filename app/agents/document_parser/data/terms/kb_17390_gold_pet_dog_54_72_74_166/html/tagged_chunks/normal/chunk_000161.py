from langchain_core.documents import Document

chunk = Document(
    page_content=('체결한 경우 45일)이 초과된 계약은 청약을 철회할 수 없습니다.<br>\uf000 청약철회는 계약자가 전화로 신청하거나, 철회의사를 '
 '표시하기 위한 서면, 전자우<br>편, 휴대전화 문자메시지 또는 이에 준하는 전자적 의사표시(이하 "서면 등"이라<br>합니다)를 발송한 '
 '때 효력이 발생합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000161',
              'chunk_char_len': 158,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
