from langchain_core.documents import Document

chunk = Document(
    page_content=('보험금을 합산한 금액이 1인당 “1억원까지”보호됩니다. 다만, 보험계약자 및 보험료납\n'
 '부자가 법인인 보험계약의 경우에는 보호되지 않습니다.- 21 -메리츠 마음든든 반려동물보험 특별약관반려견 배상책임 특별약관제1조(목적)이 '
 '특별약관은 계약자와 회사 사이에 피보험자가 법률상의 배상책임을 부담함으로써 입은\n'
 '손해에 대한 위험을 보장하기 위하여 체결됩니다.제2조(용어의 정의)① 이 특별약관에서 사용되는 용어의 정의는 다음과 같습니다.가. '
 '배상책임: 보험증권상의 보장지역 내에서 보험기간중에 발생된 보험사고로 인하여'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000116',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
