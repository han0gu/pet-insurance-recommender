from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 용 어 풀 이 공제계약 유사보험으로서 공제 사업을 실시하는 경영주체와 공제 계약자 사이에 체결되 | 용 어 풀 이 공제계약 '
 '유사보험으로서 공제 사업을 실시하는 경영주체와 공제 계약자 사이에 체결되 |\n'
 '는 계약을 말합니다. 우체국, 신협, 새마을금고 등이 공제계약을 취급합니다.상해제6조(특별약관의 소멸)\n'
 '\uf000 회사는 제1조(보험금의 지급사유)에서 정한 반려동물장례비용지원금을 지급한 경\n'
 '우에는 그 지급사유가 발생한 때부터 이 특별약관 계약은 소멸되며 이 특별약관의 해약환급금을 지급하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000652',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
