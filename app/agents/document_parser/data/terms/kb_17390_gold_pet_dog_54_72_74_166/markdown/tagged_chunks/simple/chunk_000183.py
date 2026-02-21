from langchain_core.documents import Document

chunk = Document(
    page_content=('제34조(해약환급금) 제1항에 의한 해약환급금을 계약자에게 지급합니다.- \n'
 '# 제34조(해약환급금)해약환급금은"보험료 및 해약환급금 산출방법서"에 따라 계산합니# \uf000 이 약관에 따른- \n'
 '다.\n'
 '\uf000 해약환급금의 지급사유가 발생한 경우 계약자는 회사에 해약환급금을 청구하여야- \n'
 '- 68 -- \n'
 '하며, 회사는 청구를 접수한 날부터 3영업일 이내에 해약환급금을 지급합니다. 해- 약환급금 지급일까지의 기간에 대한 이자의 계산은 '
 '"보험금을 지급할 때의 적립이\n'
 '- 율 계산"(【별표2】참조)에 따릅니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000183',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
