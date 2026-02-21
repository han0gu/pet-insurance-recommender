from langchain_core.documents import Document

chunk = Document(
    page_content=('- 향후 관련법령이 개정된 경우 개정된 내용을 적용합니다.\n'
 '※\uf000제1항에도 불구하고 청약한 날부터 30일(단, 만 65세 이상의 계약자가통신수단 중- 62 -- 전화를 이용하여 체결한 경우 '
 '45일)이 초과된 계약은 청약을 철회할 수 없습니다.\n'
 '- \uf000 청약철회는 계약자가 전화로 신청하거나, 철회의사를 표시하기 위한 서면, 전자우\n'
 '- 편, 휴대전화 문자메시지 또는 이에 준하는 전자적 의사표시(이하 "서면 등"이라\n'
 '- 합니다)를 발송한 때 효력이 발생합니다. 계약자는 서면 등을 발송한 때에 그 발'),
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
 'indexing': {'chunk_id': 'chunk_000108',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
