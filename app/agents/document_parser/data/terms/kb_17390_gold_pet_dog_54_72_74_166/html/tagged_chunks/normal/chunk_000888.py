from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3호 및 제4호의 내용에 관한 사항을 계약자에게 안내할 것<br>\uf000 제1항에 따라 특별약관이 해지된 경우에는 보통약관 '
 "제1절 제34조(해약환급금)</p><br><table id='63' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>제1항에 따른</td><td>해약환급금을 "
 '계약자에게 지급합니다.</td></tr><tr><td colspan="2">용 어 풀 이 납입최고(독촉) 약정된 기일까지 보험료가 '
 '납입되지 않을 경우, 회사가 계약자에게'),
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
 'indexing': {'chunk_id': 'chunk_000888',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
