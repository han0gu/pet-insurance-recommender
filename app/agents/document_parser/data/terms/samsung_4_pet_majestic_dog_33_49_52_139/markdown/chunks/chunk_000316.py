from langchain_core.documents import Document

chunk = Document(
    page_content=('정하는 응급의료기관(권역응급의료센터, 전문응급의료센터, 지역응급의료센터, 지역응\n'
 '급의료기관) 또는 응급의료에 관한 법률 제35조의2(응급의료기관 외의 의료기관)에서\n'
 '정하는 시장 · 군수 · 구청장에게 신고되고 그 신고가 수리된 응급의료시설을 말합니\n'
 '다.\n'
 '② 관련 법령이 개정된 경우 개정된 내용을 적용하며, 의료기관의 「응급실」 해당여부는\n'
 '내원 당시 기준으로 판단합니다.- \n'
 '# 제5조 (보험금을 지급하지 않는 사유)① 회사는 다음 중 어느 한 가지로 보험금 지급사유가 발생한 때에는 보험금을 지급하지'),
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
