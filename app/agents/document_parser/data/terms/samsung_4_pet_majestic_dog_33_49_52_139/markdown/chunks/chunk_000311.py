from langchain_core.documents import Document

chunk = Document(
    page_content=('- 라 합니다) 중에 응급실에 내원하여 「아나필락시스(anaphylaxis)」 (이하 「아나필락\n'
 '- 시스」 라 합니다)로 진단확정된 경우 연간1회에 한하여 보험증권에 기재된 이 특별약\n'
 '- 관의 보험가입금액을 응급의료 아나필락시스 진단비(연간1 회한)로 보험수익자에게 지\n'
 '- 급합니다.\n'
 '- ② 제1항의 「아나필락시스」 는 제3조(아나필락시스의 정의 및 진단확정)에서 정한 아나\n'
 '- 필락시스쇼크를, 「응급실」 은 제4조(응급실의 정의)에 해당되는 의료기관을 말합니\n'
 '- 다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
