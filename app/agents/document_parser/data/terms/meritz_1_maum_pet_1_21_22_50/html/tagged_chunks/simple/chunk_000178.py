from langchain_core.documents import Document

chunk = Document(
    page_content=('보험연도에 속하는 보험료는 전액을 돌려드립니다.<br>③ 계약의 무효, 효력상실, 해지 또는 소멸로 인하여 회사가 환급하여야 할 보험료가 '
 '있을<br>때에는 계약자는 환급금을 청구하여야 하며, 회사는 청구일의 다음 날부터 지급일까지<br>의 기간에 대하여 ‘보험개발원이 '
 "공시하는 보험계약대출이율’을 연단위 복리로 계산한<br>금액을 더하여 지급합니다.</p><br><p id='95' "
 "data-category='paragraph' style='font-size:14px'>【설명】보험사가 해지권을 행사하는 경우 위의 "
 '‘청구일’은 보험사의'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000178',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
