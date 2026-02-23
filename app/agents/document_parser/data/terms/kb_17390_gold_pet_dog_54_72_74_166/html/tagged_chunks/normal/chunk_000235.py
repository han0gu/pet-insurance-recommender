from langchain_core.documents import Document

chunk = Document(
    page_content=('행하여진 경우에도 자동대출 납입전 납입최고(독촉)기<br>간이 끝나는 날의 다음날부터 1개월 이내에 계약자가 계약의 해지를 청구한 '
 '때에는<br>회사는 보험료의 자동대출 납입이 없었던 것으로 하여 제34조(해약환급금) 제1항<br>에 따른 해약환급금을 '
 '지급합니다.<br>\uf000 회사는 자동대출납입이 종료된 날부터 15일 이내에 자동대출납입이 종료되었음을<br>서면, 전화(음성녹음) '
 "또는 전자문서(SMS 포함) 등으로 계약자에게 안내하여 드립<br>니다.</p><br><h1 id='45' "
 "style='font-size:14px'>용 어 풀"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000235',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
