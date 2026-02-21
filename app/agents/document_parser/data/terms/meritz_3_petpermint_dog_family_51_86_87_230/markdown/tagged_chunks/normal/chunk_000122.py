from langchain_core.documents import Document

chunk = Document(
    page_content=('따라 계약이 효력을 잃는 경우에 회사는 제35조(해약환급\n'
 '금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.# 제35조(해약환급금)\uf000 이 약관에 따른 해약환급금은「보험료 및 '
 '해약환급금 산\n'
 '출방법서」에 따라 계산합니다.\n'
 '\uf000 해약환급금의 지급사유가 발생한 경우 계약자는 회사에\n'
 '해약환급금을 청구하여야 하며, 회사는 청구를 접수한 날부\n'
 '터 3영업일 이내에 해약환급금을 지급합니다. 해약환급금\n'
 '지급일까지의 기간에 대한 이자의 계산은【별표1(보험금을\n'
 '지급할 때의 적립이율 계산)】에 따릅니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000122',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
